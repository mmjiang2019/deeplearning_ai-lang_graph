# deeplearning_ai-lang_graph
说明：
对于项目中的代码，有一些地方需要注意：
1. 模型效果
对于本地模型，因为很多情况下由于设备性能有限，都是部署的精简化模型，所以效果可能不如官方模型。
因此导致项目中关于 langmem 相关的处理，可能不如预期效果好。

2. 关于 langmem 的使用
2.a 初始化 memory 之后，在后续代码中，只要在 runnable context 中，就可以直接使用 store 来获取记忆。
    或者可以通过 store = langgraph.config.get_store() 来获取。
2.b config 同样属于 runnable context 的一部分，所以在 config 中也可以直接用
2.c state 同上