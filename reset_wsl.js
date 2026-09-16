module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "rm -rf app/env app/.installed_wsl"
      }
    },
    {
      method: "input",
      params: {
        title: "Reset completed",
        description: "WSL environment reset. Click Install to reinstall."
      }
    }
  ]
}
