# 运行前准备

需要配置好Docker环境，安装好Docker Compose

目前压缩文件中包含了前端chatbot-client和后端chatbot-server。前端chatbot-client使用Vue.js开发，后端chatbot-server使用Python Django开发。

其中两个文件中都设置了相应的Docker镜像构建文件，运行命令如下所述。

此外，运行需要基于sqlite，目前已经默认配置了一个，并且存储了API key。运行后即可通过admin界面做功能上的配置。

由于部分功能没有用到，所以对于不相关功能尽量保持默认设置，如果需要使用，可以根据代码进行相关配置开启。

# 运行步骤

## 1. 构建前端和后端共三个镜像

### a. 前端 

进入 `chatbot-client` 目录，运行命令：

```bash
docker build -t chatbot:latest .
```

可以得到一个名为 `chatbot:latest` 的镜像。


如果遇到依赖问题，使用如下命令安装依赖

```bash
nvm install 18.0.0
nvm use 18.0.0
node -v
yarn install
```

### b. 后端

进入 `chatbot-server` 目录，运行命令：

```bash
docker-compose build
```

可以得到两个镜像，分别为 `chatbot-web-server` 和 `chatbot-wsgi-server`。后者作用是为前者提供wsgi服务。

如果遇到python依赖问题，使用如下命令安装依赖

```bash
pip install -r requirements.txt
```

## 2. 运行

进入 `chatbot-client` 目录，运行命令：

```bash
docker-compose up
```

## 3. 使用

在数据库中已经预先设定了API key，可以直接使用。

在浏览器中输入 `http://localhost` 即可访问。

已预置管理员：

默认超级用户: admin

默认密码: password


输入 `http://localhost:9000/admin` 进入管理面板，使用上述管理员密码即可。




Service Name	       Local Machine Port	Docker Service     Nginx
wsgi-server	           8000	                8000	      
web-server	           9000	                80	            
client	               CLIENT_PORT (800)	80	               Nginx 80/443 
backend-wsgi-server	   WSGI_PORT (8000)	    8000	     
backend-web-server	   SERVER_PORT (9000)	80	         
