class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.visited = False
        self.grad = None

    def add_child(self, child):
        self.children.append(child)

def dfs_topological_sort(node, sorted_list):
    node.visited = True
    for child in node.children:
        if not child.visited:
            dfs_topological_sort(child, sorted_list)
    sorted_list.append(node)

def backward(node):
    sorted_list = []
    dfs_topological_sort(node, sorted_list)
    sorted_list.reverse()
    for n in sorted_list:
        print(f"Computing gradient for node: {n.name}")
        # 在实际的实现中，这里会计算梯度并传递给前面的节点
        n.grad = 1  # 假设梯度为 1

# 构建一个简单的计算图
a = Node("a")
b = Node("b")
c = Node("c")
d = Node("d")
e = Node("e")

a.add_child(b)
a.add_child(c)
b.add_child(d)
c.add_child(d)
d.add_child(e)

# 反向传播从节点 e 开始
backward(e)

# 打印每个节点的梯度
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")
print(f"c.grad: {c.grad}")
print(f"d.grad: {d.grad}")
print(f"e.grad: {e.grad}")
