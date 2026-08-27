---
title: "Docker 笔记"
tags: [docker, container, devops, tool]
date: 2026-08-27
---


## 📖 概述：Docker 是什么？

简单来说，**Docker 是一个开源的容器化平台**。它允许你将应用程序及其所有依赖（如库、配置文件、环境变量）打包成一个标准化的单元，称为**容器**。

这个容器可以轻松地在任何装有 Docker 的服务器上运行，从而解决了著名的环境不一致问题：“**在我电脑上能运行，为什么到你电脑上就不行了？**”。

**它能干什么？**
- **环境标准化**：告别“环境不一致”的噩梦，确保开发、测试、生产环境高度一致。
- **快速部署与启动**：容器是轻量级的，启动时间仅为毫秒级，远比传统虚拟机快。
- **资源高效隔离**：容器共享宿主机内核，在提供资源隔离的同时，开销远小于虚拟机。

## 🏗️ 核心概念

理解这三个核心概念是掌握 Docker 的基石：

1.  **镜像 (Image)**：一个只读的**模板**，包含了运行一个应用程序所需的代码、运行时、库、环境变量和配置文件。你可以把它想象成面向对象编程中的“类”。
2.  **容器 (Container)**：镜像是“类”，容器就是“实例”——**镜像的运行实体**。你可以创建、启动、停止、移动或删除一个容器。
3.  **仓库 (Registry)**：一个**存储和分发镜像**的地方，就像应用商店。最著名的公共仓库是 [Docker Hub](https://hub.docker.com/)。

## 🛠️ 环境准备：安装与验证

在开始操作前，请根据你的操作系统安装 Docker：
- **Windows / macOS**：推荐安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)，它提供了图形化界面和完整的运行环境。
- **Linux (如 CentOS/Ubuntu)**：使用包管理器安装，如 `sudo apt install docker.io`。

安装完成后，打开终端（命令行）执行以下命令验证是否成功：
```bash
docker run hello-world
```
如果看到欢迎信息，说明 Docker 已正确安装并可以运行！

## 🚀 核心操作：动手实践

> **注意**：所有命令均在终端中执行。

### 1. 镜像操作 (Image Operations)

| 操作 | 命令示例 | 说明 |
| :--- | :--- | :--- |
| **拉取镜像** | `docker pull nginx:latest` | 从仓库下载名为 `nginx`、标签为 `latest` 的镜像到本地 |
| **列出镜像** | `docker images` | 查看本机已下载的所有镜像 |
| **删除镜像** | `docker rmi nginx:latest` | 删除指定的镜像 |

### 2. 容器操作 (Container Operations)

这是最常用的部分，请重点关注。

| 操作 | 命令示例 | 说明 |
| :--- | :--- | :--- |
| **创建并启动容器** | `docker run -d --name my-nginx -p 8080:80 nginx` | **最核心命令**。从 `nginx` 镜像创建并启动一个名为 `my-nginx` 的容器。<br>• `-d`：后台运行<br>• `--name`：指定容器名<br>• `-p 8080:80`：将宿主机的 `8080` 端口映射到容器的 `80` 端口 |
| **列出运行中的容器** | `docker ps` | 查看当前正在运行的容器 |
| **停止容器** | `docker stop my-nginx` | 停止运行中的容器 |
| **启动已停止的容器** | `docker start my-nginx` | 重新启动一个已停止的容器 |
| **进入容器内部** | `docker exec -it my-nginx bash` | 进入运行中容器的命令行，进行调试或配置 |
| **查看容器日志** | `docker logs my-nginx` | 查看容器的标准输出日志，用于排查问题 |
| **删除容器** | `docker rm my-nginx` | 删除容器（**需先停止**） |

**实践任务**：执行 `docker run -d --name my-nginx -p 8080:80 nginx` 后，在浏览器访问 `http://localhost:8080`，你应该能看到 Nginx 的欢迎页！这就是你成功运行第一个 Web 服务器的证明。

### 3. 创建自定义镜像 (Using Dockerfile)

当你需要打包自己的应用时，就需要编写 `Dockerfile`。
1.  创建一个名为 `Dockerfile` 的文件。
2.  写入以下内容（以 Python Flask 应用为例）：
    ```dockerfile
    # 指定基础镜像
    FROM python:3.9-slim
    # 设置工作目录
    WORKDIR /app
    # 复制依赖文件并安装
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    # 复制应用代码
    COPY . .
    # 声明容器运行时监听的端口
    EXPOSE 5000
    # 指定启动命令
    CMD ["python", "app.py"]
    ```
3.  在 `Dockerfile` 所在目录执行 `docker build -t my-flask-app .` 构建镜像。
4.  最后用 `docker run -d -p 5000:5000 my-flask-app` 运行你的应用。

## 📚 参考资料

-   **官方文档 (必读)**：[Docker 官方文档](https://docs.docker.com/) 是最权威、最全面的信息来源。
-   **开源教程 (体系化学习)**：
    -   [Docker — 从入门到实践](https://yeasy.gitbook.io/docker_practice/)：非常经典的在线教程，内容由浅入深。
    -   [datawhalechina/docker-notes](https://github.com/datawhalechina/docker-notes)：一个结构清晰的 Docker 学习项目，包含大量实例。
-   **GitHub 仓库 (参考他人笔记)**：
    -   [docker-from-zero-to-hero](https://github.com/ritesh355/docker-from-zero-to-hero)：一个从安装到实战的 step-by-step 指南。
    -   [docker-tutorial](https://github.com/ch840120/docker-tutorial)：一个中文 Docker 教学仓库，内容组织清晰。
