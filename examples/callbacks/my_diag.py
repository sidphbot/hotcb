class MyDiag:
    def __init__(self, id: str, msg: str = "hi"):
        self.id = id
        self.msg = msg
    def set_params(self, **kwargs):
        if "msg" in kwargs: self.msg = kwargs["msg"]
    def handle(self, event, env):
        step = int(env.get("step", 0))
        if step % 30 == 0:
            env.get("log", print)(f"[{self.id}] {self.msg} step={step} event={event}")