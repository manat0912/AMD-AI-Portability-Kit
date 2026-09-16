module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "pip install gradio colorama torch"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click the 'Start' button to launch the AMD AI Portability Kit."
      }
    }
  ]
}
