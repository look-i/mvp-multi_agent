# 导入必要的模块
from fastapi import FastAPI, HTTPException, Request, Body, File, UploadFile, APIRouter, Query, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import uuid
import shutil
import logging
import random

# LangChain, AutoGen, 和 Supabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from supabase.client import Client, create_client
from openai import OpenAI # 直接使用OpenAI客户端，因为AutoGen需要它
from gotrue.errors import AuthApiError

# --- 配置 ---

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从环境变量读取API密钥和Supabase配置
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # 使用Service Key，因为它有权写入数据库

if not all([MOONSHOT_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise ValueError("环境变量 MOONSHOT_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY 必须全部设置！")

# --- 初始化客户端 ---

# 初始化 FastAPI 应用
app = FastAPI(title="AI教育多-智能体系统 (Supabase版)", description="基于AutoGen和Supabase的无状态多智能体系统")

# 配置 CORS 中间件
# 这解决了前端访问后端时的跨域问题
origins = [
    "https://mvp-frontend-ln39.vercel.app",  # 允许您的 Vercel 前端
    "http://localhost:5173",              # 允许本地开发环境
    "http://127.0.0.1:5173",             # 允许本地开发环境
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 头部
)

# 创建一个全局的、拥有服务权限的Supabase客户端，用于执行需要管理员权限的操作（例如上传知识库）
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
logger.info("Supabase 管理员客户端初始化成功")

# 初始化 OpenAI 客户端 (用于直接调用，以适配无状态模式)
openai_client = OpenAI(
    api_key=MOONSHOT_API_KEY,
    base_url="https://api.moonshot.cn/v1"
)

# 初始化 LangChain 嵌入模型
embeddings_model = OpenAIEmbeddings(
    api_key=MOONSHOT_API_KEY,
    base_url="https://api.moonshot.cn/v1",
    model="moonshot-v1-embedding"
)

# --- 认证依赖 ---

auth_scheme = HTTPBearer()

async def get_current_user(token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """验证JWT并返回用户信息"""
    try:
        # 使用 Supabase GoTrue 客户端验证 token
        user_response = supabase_admin.auth.get_user(token.credentials)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=401, detail="无效的用户凭证")
        return user
    except AuthApiError as e:
        logger.error(f"JWT 验证失败: {e}")
        raise HTTPException(status_code=401, detail=f"认证失败: {e}")
    except Exception as e:
        logger.error(f"处理认证时发生未知错误: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

# --- 数据模型 ---

class ConversationCreateRequest(BaseModel):
    user_id: str
    conversation_type: str = "student_self_study"
    title: Optional[str] = "新的对话"

class MessageCreateRequest(BaseModel):
    content: str
    use_rag: bool = True

class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

# --- 智能体角色定义 ---
# 我们不再创建和存储AutoGen的实例，而是将它们定义为包含角色的“系统提示词”模板
AGENT_ROLES = {
    "expert": {
        "name": "专家智能体",
        "system_message": """你是一位人工智能教育专家，负责制定学习目标和评估标准。
        你的职责包括：
        1. 根据学生的学习历史和能力水平，动态调整任务的难度和评价维度。
        2. 将专业的考核标准转化为学生能理解的、可自评的学习评估量规。
        3. 确保学习目标符合《广东省中小学学生人工智能素养框架》的四大维度。
        你的回答风格应该是苏格拉底式的，通过提问引导学生思考，而不是直接给出答案。"""
    },
    "assistant": {
        "name": "助教智能体",
        "system_message": """你是一位人工智能教育助教，负责提供学习资源和任务指导。
        你的职责包括：
        1. 提供多元化的学习资源，如文本、活动、在线模拟器或视频。
        2. 将复杂的任务分解为清晰的子步骤，提供清晰的行动路线图。
        3. 基于专家制定的量规，逐条进行评价，给予有据可依的反馈。
        4. 提出启发式问题，引导学生进行深度思考。
        你的回答风格应该是苏格拉底式的，通过提问引导学生思考。"""
    },
    "peer": {
        "name": "同伴智能体",
        "system_message": """你是一位人工智能学习同伴，与学生一起学习。
        你的职责包括：
        1. 在完成任务时，展示你的思考过程，让学生了解问题解决的思路。
        2. 故意犯一些初学者典型的错误，然后进行自我修正，帮助学生理解常见错误。
        3. 以平等的姿态与学生交流，营造轻松友好的学习氛围。
        你的回答风格应该是苏格拉底式的，通过提问引导学生思考。"""
    }
}

# --- 辅助函数 ---

def get_relevant_documents_from_db(query_text: str, top_k: int = 3) -> tuple[str, list]:
    """从Supabase数据库中检索相关文档，并返回上下文和引用"""
    try:
        query_embedding = embeddings_model.embed_query(query_text)
        # 使用管理员客户端进行文档匹配
        res = supabase_admin.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_threshold': 0.7,
            'match_count': top_k
        }).execute()

        if not res.data:
            return "", []

        context = "\n".join([doc['content'] for doc in res.data])
        references = [{"source": doc['metadata'].get('source', '未知来源'), "content": doc['content']} for doc in res.data]
        logger.info(f"已从数据库检索到 {len(res.data)} 条相关内容")
        return context, references
    except Exception as e:
        logger.error(f"从数据库检索文档时出错: {e}")
        return "检索知识库时发生错误", []

# --- API 路由 ---

api_router = APIRouter()


@api_router.get("/")
async def root():
    return {"message": "AI教育多智能体系统API (Supabase版)"}


@api_router.get("/conversations")
async def get_conversations(user: dict = Depends(get_current_user)):
    """根据用户ID获取对话列表"""
    try:
        # user.id 是从验证过的JWT中获取的，确保了安全性
        res = supabase_admin.table('conversations').select('*').eq('user_id', user.id).order('created_at', desc=True).execute()
        return res.data
    except Exception as e:
        logger.error(f"获取对话列表时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/conversations", status_code=201)
async def create_conversation(req: ConversationCreateRequest, user: dict = Depends(get_current_user)):
    """创建一次新的对话会话"""
    try:
        # 强制使用已验证的用户ID
        user_id = user.id
        res = supabase_admin.table('conversations').insert({
            'user_id': user_id,
            'conversation_type': req.conversation_type,
            'title': req.title,
            'agent_roles_involved': list(AGENT_ROLES.keys())
        }).execute()
        
        if not res.data:
            raise HTTPException(status_code=500, detail="创建对话失败")
            
        return res.data[0]
    except Exception as e:
        logger.error(f"创建对话时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def create_message(conversation_id: str, req: MessageCreateRequest, user: dict = Depends(get_current_user)):
    """处理用户消息，并获取一个智能体的回复"""
    try:
        # 验证用户是否有权访问此对话
        conv_res = supabase_admin.table('conversations').select('user_id').eq('id', conversation_id).single().execute()
        if not conv_res.data or conv_res.data['user_id'] != user.id:
            raise HTTPException(status_code=403, detail="无权访问此对话")

        # 1. 将用户的消息存入数据库
        user_message_res = supabase_admin.table('messages').insert({
            'conversation_id': conversation_id,
            'role': 'user',
            'content': req.content
        }).execute()
        if not user_message_res.data:
            raise HTTPException(status_code=500, detail="存储用户消息失败")

        # 2. 从数据库获取完整的历史消息
        history_res = supabase_admin.table('messages').select('*').eq('conversation_id', conversation_id).order('created_at', desc=False).execute()
        history = history_res.data or []
        
        # 3. 随机选择一个智能体角色来回复
        agent_role_key = random.choice(list(AGENT_ROLES.keys()))
        agent_profile = AGENT_ROLES[agent_role_key]
        
        # 4. 如果需要，从知识库检索上下文
        rag_context, references = "", []
        if req.use_rag:
            rag_context, references = get_relevant_documents_from_db(req.content)

        # 5. 构建发送给大模型的聊天消息列表
        messages_for_api = [{"role": "system", "content": agent_profile["system_message"]}]
        if rag_context:
            messages_for_api[0]["content"] += f"\n\n请参考以下资料：\n{rag_context}"
            
        for msg in history:
            # 确保角色是 'user' 或 'assistant'
            api_role = 'assistant' if msg['role'] != 'user' else 'user'
            messages_for_api.append({"role": api_role, "content": msg['content']})

        # 6. 调用OpenAI API获取回复
        response = openai_client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=messages_for_api,
            temperature=0.7
        )
        agent_response_content = response.choices[0].message.content

        # 7. 将智能体的回复存入数据库
        agent_message_res = supabase_admin.table('messages').insert({
            'conversation_id': conversation_id,
            'role': agent_role_key, # 存储实际的智能体角色
            'content': agent_response_content,
            'metadata': {'references': references}
        }).execute()
        if not agent_message_res.data:
            raise HTTPException(status_code=500, detail="存储智能体回复失败")

        return agent_message_res.data[0]

    except Exception as e:
        logger.error(f"处理消息时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/knowledge/upload", status_code=201)
async def upload_knowledge(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """上传知识库文件，处理并存入Supabase数据库"""
    # 简单的权限检查：可以设计更复杂的逻辑，例如只允许教师角色上传
    if not user:
         raise HTTPException(status_code=403, detail="只有登录用户才能上传知识库")

    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        documents_to_insert = []
        for text in texts:
            embedding = embeddings_model.embed_query(text.page_content)
            documents_to_insert.append({
                'content': text.page_content,
                'metadata': {'source': file.filename, 'uploader_id': user.id},
                'embedding': embedding
            })
        
        supabase_admin.table('documents').insert(documents_to_insert).execute()
        shutil.rmtree(temp_dir)

        return {"message": f"文件 {file.filename} 已成功处理", "chunks_added": len(documents_to_insert)}
    except Exception as e:
        logger.error(f"上传知识库文件时出错: {e}")
        if os.path.exists("temp_uploads"):
            shutil.rmtree("temp_uploads")
        raise HTTPException(status_code=500, detail=f"上传知识库文件失败: {e}")

app.include_router(api_router)


# 用于本地调试
if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
