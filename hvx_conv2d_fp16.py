import tvm
from tvm import te
import tvm.testing
import numpy as np
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple, traverse_inline
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.nn.pad import pad
from tvm.topi.transform import layout_transform

pytest_plugins = [
    "tvm.contrib.hexagon.pytest_plugin",
]


def conv2d_NCHWc(
    data, kernel, stride, padding, dilation, out_dtype="float16"
):
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    )

    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = get_const_tuple(
        kernel.shape
    )

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)

    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD:
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data

    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    return te.compute(
        oshape,
        lambda n, oc_chunk, oh, ow, oc_block: te.sum(
            data_pad[
                n,
                idxdiv(ic, ic_bn),
                oh * HSTR + kh * dilation_h,
                ow * WSTR + kw * dilation_w,
                idxmod(ic, ic_bn),
            ].astype(out_dtype)
            * kernel[oc_chunk, idxdiv(ic, ic_bn), kh, kw, idxmod(ic, ic_bn), oc_block].astype(
                out_dtype
            ),
            axis=[ic, kh, kw],
        ),
        name="conv2d_NCHWc",
        tag="conv2d_NCHWc",
    )


def schedule_conv2d_NCHWc(outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_NCHWc" in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = (
                data_vec.op.input_tensors[0]
                if isinstance(data_vec.op, te.tensor.ComputeOp) and "pad" not in data_vec.op.tag
                else data_vec
            )
            if isinstance(data.op, te.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            args = [s, data_vec, conv_out, outs[0]]
            # int8 conv kernel is 7-dim
            _, _, kh, kw, _, _, = get_const_tuple(kernel_vec.shape)
            # assert n_elems == 4
            dtype = "uint" if data.dtype == "uint8" else "int"

            inline_fused = True

            out_width = conv_out.shape[3]
            reg_n = 1
            for n in range(31, 0, -1):
                if out_width % n == 0:
                    reg_n = n
                    break

            schedule_conv_NCHWc_cpu_common(
                *args, reg_n=reg_n, inline_fused=inline_fused
            )

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_conv_NCHWc_cpu_common(
    s,
    data_vec,
    conv_out,
    last,
    reg_n,
    inline_fused=True,
):
    unroll_kw = False
    _, _, _, _, ic_bn = get_const_tuple(data_vec.shape)
    _, _, _, _, oc_bn = get_const_tuple(conv_out.shape)

    # schedule pad
    if isinstance(s[data_vec].op, te.tensor.ComputeOp) and "pad" in data_vec.op.tag:
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        # s[data_vec].parallel(parallel_axis)
        data_vec = data_vec.op.input_tensors[0]

    # schedule 5-D NCHW[x]c conv
    C, O = conv_out, last
    CC = s.cache_write(C, "global")

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(batch, oc_chunk, oh)
    s[C].vectorize(oc_block)

    if C == O:
        s[C].parallel(parallel_axis)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis

    ic, kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)
    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    if unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, kw, ic_block, ow_block, oc_block)

    s[CC].vectorize(oc_block)
    s[CC].unroll(ow_block)

    return s


# @tvm.testing.requires_hexagon
# def test_conv2d_f16f16f16(hexagon_session):
dtype = "float16"
ic_bn = 64
oc_bn = 64
I = 64
O = 64
H = 56
W = 56
kH = 3
kW = 3

data_packed = te.placeholder((1, I // ic_bn, H, W, ic_bn), name="data", dtype=dtype)
kernel_packed = te.placeholder((O // oc_bn, I // ic_bn, kH, kW, ic_bn, oc_bn), dtype=dtype)

strides = (1, 1)
padding = (1, 1)
dilation = (1, 1)

out = conv2d_NCHWc(
    data_packed, kernel_packed, strides, padding, dilation
)

s = schedule_conv2d_NCHWc([out])

target = "llvm --device arm_cpu --mtriple aarch64-apple-darwin"
module = tvm.build(s, [data_packed, kernel_packed, out], target)
dev = tvm.cpu(0)

# target_hexagon = tvm.target.hexagon("v69", link_params=True)
# target = tvm.target.Target(target_hexagon, host=target_hexagon)
# f = tvm.build(s, [data_packed, kernel_packed, out], target)
# module = hexagon_session.load_module(f)
# dev = hexagon_session.device

a_np = np.random.randn(1, I, H, W).astype("float16")
w_np = np.random.randn(O, I, kH, kW).astype("float16")
c_np = tvm.topi.testing.conv2d_nchw_python(a_np.astype("float32"), w_np.astype("float32"), strides, padding)

packed_data_np = np.zeros(get_const_tuple(data_packed.shape)).astype(dtype)
packed_w_np = np.zeros(get_const_tuple(kernel_packed.shape)).astype(dtype)

for i in range(I):
    for h in range(H):
        for w in range(W):
            packed_data_np[0, i // ic_bn, h, w, i % ic_bn] = a_np[0, i, h, w]

for o in range(O):
    for i in range(I):
        for h in range(kH):
            for w in range(kW):
                packed_w_np[o // oc_bn, i // ic_bn, h, w, (i % ic_bn), o % oc_bn] = w_np[o, i, h, w]



a = tvm.nd.array(packed_data_np.astype(dtype), dev)
w = tvm.nd.array(packed_w_np.astype(dtype), dev)

c = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=out.dtype), dev)

module(a, w, c)

P, Q = c_np.shape[2:4]

evaluator = module.time_evaluator(module.entry_name, dev, number=20)
time_ms = evaluator(a, w, c).mean * 1e3
gflops = (O * P * Q * I * kH * kW) * 2 / 1e9
print("time elapsed: ", time_ms)
print("GFLOPS:", gflops / (time_ms / 1e3))

out_packed = c.numpy()

out = np.zeros(c_np.shape).astype("float16")

for o in range(O):
    for h in range(P):
        for w in range(Q):
            out[0, o, h, w] = out_packed[0, o // oc_bn, h, w, o % oc_bn]

print(np.max(np.abs(out - c_np)), np.mean(np.abs(out - c_np)))

mx = np.max(np.abs(out - c_np))

indices = np.where(np.abs(out - c_np) == mx)

print(out[indices], c_np[indices])

# test_conv2d_f16f16f16(None)
