import logging

from jinja2 import (
    Environment,
    PackageLoader,
    Template,
    TemplateNotFound,
    select_autoescape,
)

logger = logging.getLogger(__name__)


class PromptLoader:
    def __init__(self) -> None:
        self.env = Environment(
            loader=PackageLoader("model", "templates"),
            autoescape=select_autoescape(),
        )

    def list_templates(self) -> list[str]:
        """List all available templates."""
        return self.env.list_templates()

    def load_template(self, file_path: str) -> Template:
        """Load a Jinja2 template from a file."""
        try:
            template = self.env.get_template(file_path)
            logger.debug(f"Loaded template: {file_path}")
            return template
        except TemplateNotFound as e:
            logger.error(f"Error loading template {file_path}: {e}")
            raise

    def render_template(self, file_path: str, template_variables: dict) -> str:
        """Render a Jinja2 template to a string."""
        try:
            template = self.load_template(file_path)
            rendered_prompt = template.render(**template_variables)
            logger.debug(
                f"Rendered template {file_path} with variables {template_variables}"
            )
            return rendered_prompt
        except Exception as e:
            logger.error(
                f"Error rendering template {file_path} with variables {template_variables}: {e}"
            )
            raise
