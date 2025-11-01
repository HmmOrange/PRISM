from prism.actions.action_node import ActionNode

class ActionGraph:
    """ActionGraph: a directed graph to represent the dependency between actions."""

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.execution_order = []
        self.context = {}  # Lưu kết quả thực thi các node
        self.overall_task = ""
        
    def set_overall_task(self, overall_task: str):
        self.overall_task = overall_task

    def add_node(self, node):
        """Add a node to the graph"""
        self.nodes[node.key] = node

    def add_edge(self, from_node: ActionNode, to_node: ActionNode):
        """Add an edge to the graph"""
        if from_node.key not in self.edges:
            self.edges[from_node.key] = []
        self.edges[from_node.key].append(to_node.key)
        from_node.add_next(to_node)
        to_node.add_prev(from_node)

    def topological_sort(self):
        """Topological sort the graph"""
        visited = set()
        stack = []

        def visit(k):
            if k not in visited:
                visited.add(k)
                if k in self.edges:
                    for next_node in self.edges[k]:
                        visit(next_node)
                stack.insert(0, k)

        for key in self.nodes:
            visit(key)

        self.execution_order = stack

    async def execute(self):
        print("\nBắt đầu thực thi action graph:")
        for key in self.execution_order:
            node = self.nodes[key]
            prev_context = {
                prev.key: {
                    "instruction": prev.instruction,
                    "result": self.context.get(prev.key),
                }
                for prev in node.prev_nodes
            }
            print(f"\nThực thi task {node.key}: {node.instruction}")
            print(f"Context nhận được từ các node trước: {prev_context}")
            result = await node.execute_node(self.overall_task, prev_context)
            self.context[node.key] = result
        # print("\nHoàn thành action graph!")
        # print("\nTổng kết context:")
        # for k, v in self.context.items():
        #     print(f"{k}: {v}")


def build_action_graph(tasks):
    key_to_node = {t['task_id']: ActionNode(t['task_id'], t['instruction']) for t in tasks}
    graph = ActionGraph()
    for node in key_to_node.values():
        graph.add_node(node)
    for t in tasks:
        for dep in t['dependent_task_ids']:
            graph.add_edge(key_to_node[dep], key_to_node[t['task_id']])
    return graph

if __name__ == "__main__":
    tasks = [
        {"task_id": "1", "dependent_task_ids": [], "instruction": "Load the image /images/0.png for analysis.", "task_type": "TaskType.ML_TASK"},
        {"task_id": "2", "dependent_task_ids": ["1"], "instruction": "Use an object detection model to identify and count animals (dogs and cats) in the image.", "task_type": "TaskType.ML_TASK"},
        {"task_id": "3", "dependent_task_ids": ["2"], "instruction": "Determine if the image contains only dogs, only a single cat, or a mixture of animals.", "task_type": "TaskType.LOGIC_TASK"},
        {"task_id": "4", "dependent_task_ids": ["2"], "instruction": "If only dogs are detected, identify the breeds of the dogs and find the most frequent breed.", "task_type": "TaskType.ML_TASK"},
        {"task_id": "5", "dependent_task_ids": ["2"], "instruction": "If a single cat is detected, analyze the cat's face to determine its emotion.", "task_type": "TaskType.ML_TASK"},
        {"task_id": "6", "dependent_task_ids": ["3", "4", "5"], "instruction": "Based on the detection and analysis, decide whether to return the most frequent dog breed, the cat's emotion, or 'unsure' if the image is unclear.", "task_type": "TaskType.LOGIC_TASK"}
    ]
    graph = build_action_graph(tasks)
    graph.topological_sort()
    graph.execute()
