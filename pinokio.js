const path = require('path')
module.exports = {
  version: "1.5",
  title: "AMD AI Portability Kit",
  description: "Audit AI projects for CUDA lock-in, plan DirectML/ONNX/ROCm alternatives, and convert source-available CUDA to reviewed HIP drafts.",
  icon: "icon.svg",
  menu: async (kernel, info) => {
    let installing = info.running("install.js")
    let installingWSL = info.running("install_wsl.js")
    let installed = info.exists("app/env")
    let installedWSL = info.exists("app/.installed_wsl")
    let running = info.running("start.js")
    let start_wsl = info.running("start_wsl.js")
    let update_wsl = info.running("update_wsl.js")
    let reset_wsl = info.running("reset_wsl.js")
    if (installing) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }]
    } else if (installingWSL) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing (WSL)",
        href: "install_wsl.js",
      }]
    } else if (installedWSL) {
      if (start_wsl) {
        let local = info.local("start_wsl.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI (WSL)",
            href: local.url,
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal (WSL)",
            href: "start_wsl.js",
          }]
        }
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Terminal (WSL)",
          href: "start_wsl.js",
        }]
      } else if (update_wsl) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Updating (WSL)",
          href: "update_wsl.js",
        }]
      } else if (reset_wsl) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Resetting (WSL)",
          href: "reset_wsl.js",
        }]
      }
      return [{
        default: true,
        icon: "fa-solid fa-power-off",
        text: "Start (WSL)",
        href: "start_wsl.js",
      }, {
        icon: "fa-solid fa-plug",
        text: "Update (WSL)",
        href: "update_wsl.js",
      }, {
        icon: "fa-solid fa-plug",
        text: "Install (WSL)",
        href: "install_wsl.js",
      }, {
        icon: "fa-regular fa-circle-xmark",
        text: "<div><strong>Reset (WSL)</strong><div>Revert to pre-install state</div></div>",
        href: "reset_wsl.js",
        confirm: "Are you sure you wish to reset the app?"
      }]
    } else if (installed) {
      if (running) {
        let local = info.local("start.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.js",
          }]
        }
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Terminal",
          href: "start.js",
        }]
      }
      return [{
        default: true,
        icon: "fa-solid fa-power-off",
        text: "Start",
        href: "start.js",
      }, {
        icon: "fa-solid fa-plug",
        text: "Update",
        href: "update.js",
      }, {
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }, {
        icon: "fa-regular fa-circle-xmark",
        text: "Reset",
        href: "reset.js",
      }]
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }, {
        icon: "fa-solid fa-plug",
        text: "Install (WSL)",
        href: "install_wsl.js",
      }]
    }
  }
}
