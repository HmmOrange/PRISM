from prism.prompts.prompt_template_manager import PromptTemplateManager

if __name__ == '__main__':
    prompt_template_manager = PromptTemplateManager(template_dirs=["prism/ext/mas/prompts"])
    input_message = prompt_template_manager.render(name="glue_code", passage="passage")
    print(input_message)