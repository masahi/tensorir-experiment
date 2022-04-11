import tvm

path_lib = "deploy_lib_qbert.tar"
# path_lib = "dense.tar"
dev = tvm.cpu(0)
loaded_lib = tvm.runtime.load_module(path_lib)
runtime = tvm.contrib.graph_executor.GraphModule(loaded_lib["default"](dev))
runtime.run()

print(runtime.benchmark(dev, number=1, repeat=100).mean)
