from jinja2.sandbox import SandboxedEnvironment


class ChatPrompt:
    def __init__(self, template):
        self.template = (
            SandboxedEnvironment().from_string(template) if template else None
        )

    def render(self, user_message):
        if self.template is None:
            return (
                "<|im_start|>user\n"
                + user_message
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        return self.template.render(
            messages=[{"role": "user", "content": user_message}],
            tools=None,
            add_generation_prompt=True,
            enable_thinking=False,
        )
