from model.prompt_loader import PromptLoader


class ConversationBuilder:
    def __init__(self, system_prompt_path: str):
        self.prompt_loader = PromptLoader()
        self.system_prompt = self.prompt_loader.render_template(system_prompt_path, {})
        self.initialize_conversation()

    def initialize_conversation(self):
        """Initialize the conversation with system prompt."""
        try:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        except Exception as e:
            raise RuntimeError(f"Failed to initialize conversation: {e}")

    def add_message(self, role: str, template_path: str, variables: dict):
        """Add a message to the conversation."""
        content = self.prompt_loader.render_template(template_path, variables)
        self.messages.append({"role": role, "content": content})

    def get_conversation(self) -> list:
        """Get the current conversation messages."""
        return self.messages
