module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "python3 -m venv env",
        path: "app"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "source env/bin/activate && uv pip install gradio colorama",
        path: "app"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "source env/bin/activate && uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1 --force-reinstall",
        path: "app"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "mkdir -p app && touch app/.installed_wsl"
      }
    },
    {
      method: "input",
      params: {
        title: "Installation completed",
        description: "Click Start to launch."
      }
    }
  ]
}
