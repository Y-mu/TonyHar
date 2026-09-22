class Memory:
    def __init__(self, max_turns: int = 20):
        self.messages: list[dict] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})
        self._trim()

    def _trim(self):
        # 保留 system + 最近 N 轮
        if len(self.messages) > self.max_turns * 2:
            system = [m for m in self.messages if m["role"] == "system"]
            rest = [m for m in self.messages if m["role"] != "system"]
            self.messages = system + rest[-self.max_turns * 2:]

    def get(self) -> list[dict]:
        return self.messages