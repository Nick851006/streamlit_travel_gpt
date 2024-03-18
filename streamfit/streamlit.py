# ------------------------------------------------------------------------
# 不帶記憶/歷史的Streamlit聊天 - 使用Amazon Bedrock和LangChain
# ------------------------------------------------------------------------

# import boto3  # 導入boto3模塊，用於訪問Amazon Web Services (AWS)
# import botocore  # 導入botocore模塊，botocore是AWS SDK的底層庫
from langchain_core.output_parsers import StrOutputParser  # 從langchain_core導入輸出解析器
# 從langchain_core導入聊天提示模板
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 從langchain_community導入Bedrock聊天模型
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
import openai
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessage
from prompt import classfication_type,  search_travel_recommendations_template, reply_travel_recommendations_template, other_tool_template
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
# from langchain.chat_models import ChatOpenAI as chat
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ------------------------------------------------------------------------
# Amazon Bedrock設置

# bedrock_runtime = boto3.client(
#     service_name="bedrock-runtime",  # 指定服務名稱為bedrock-runtime
#     region_name="us-east-1",  # 指定AWS區域名稱
# )

# model_kwargs =  {
#     "max_tokens_to_sample": 2048,  # 最大取樣token數
#     "temperature": 0.0,  # 溫度設定，用於控制生成的多樣性
#     "top_k": 250,  # top-k取樣
#     "top_p": 1,  # top-p取樣
#     "stop_sequences": ["\n\nHuman"],  # 停止序列
# }

# model_id = "anthropic.claude-instant-v1"  # 模型ID

# ------------------------------------------------------------------------
# LangChain鏈

# Streamlit Chat Message History
st.session_state.setdefault('chat_messages', [])
history = StreamlitChatMessageHistory(key="chat_messages")

def predict_conversation(memory, new_message=None):
    conversation = ""
    # 检查是否有新消息，并且对话记忆中没有消息时的特殊处理
    if new_message and len(memory) == 0:
        conversation += "[客户]: " + new_message + "\n"
        # memory.add_message(new_message)  # 将第一条消息加入对话记忆
    else:
        # 构建对话时，包含所有记忆中的消息
        for message in memory:
            prefix = "[客户]: " if isinstance(
                message, HumanMessage) else "[AI]: "
            conversation += prefix + message.content + "\n"
        conversation += "[客户]: " + new_message
    print(conversation)

    chat = ChatOpenAI(model=gpt_4_turbo_model, openai_api_key=openai.api_key)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        classfication_type)
    chat_prompt = ChatPromptTemplate.from_messages(
        [human_message_prompt]
    )
    predict_type = chat.invoke(
        chat_prompt.format_prompt(
            chat_history=conversation
        ).to_messages()
    )
    predict_type = predict_type.content
    print(predict_type)
    return predict_type


def process_reply(reply_text):
    """
    檢查輸入的字串中是否包含"感謝您"。
    如果包含，則返回包括"感謝您"及其之後的所有文本。
    如果不包含，則返回整個字串。

    Parameters:
    reply_text (str): 從模型或函數獲得的回覆文本。

    Returns:
    str: 根據是否包含"感謝您"，返回修改後的文本。
    """
    index = reply_text.find("感謝您")
    if index != -1:
        # 返回包含"感謝您"的位置及其之後的文本
        return reply_text[index:] + '，請稍候...'
    else:
        # 如果不包含，返回整個文本
        return reply_text

# ------------------------------------------------------------------------
# Streamlit


# Page title
st.set_page_config(page_title='Streamlit Chat')

# Clear Chat History fuction


def clear_chat_history():
    history.messages.clear()


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    openai.api_key = openai_api_key
    st.title('Streamlit Chat')
    st.subheader('With Memory :brain:')
    streaming_on = st.toggle('Streaming')
    st.button('Clear Chat History', on_click=clear_chat_history)
    st.divider()
    st.write("History Logs")
    st.write(history.messages)

gpt_4_turbo_model = "gpt-4-turbo-preview"
search_travel_template = [
    ("system", search_travel_recommendations_template),
    MessagesPlaceholder(variable_name="his"),
    ("human", "{question}"),]  # 模板設定

search_travel_prompt = ChatPromptTemplate.from_messages(
    search_travel_template)  # 從消息創建聊天提示模板
llm = ChatOpenAI(model=gpt_4_turbo_model,
                 openai_api_key=openai.api_key)

other_tool_template = [("system", other_tool_template),
                       MessagesPlaceholder(variable_name="his"),
                       ("human", "{question}"),]  # 模板設定
other_tool_prompt = ChatPromptTemplate.from_messages(
    other_tool_template)  # 從消息創建聊天提示模板

# 不帶記憶的鏈
# 將prompt、model和輸出解析器組合成一個鏈
search_travel_chain = search_travel_prompt | ChatOpenAI(model=gpt_4_turbo_model,
                                                        openai_api_key=openai.api_key) | StrOutputParser()
# 將prompt、model和輸出解析器組合成一個鏈
other_tool_chain = other_tool_prompt | ChatOpenAI(model=gpt_4_turbo_model,
                                                  openai_api_key=openai.api_key) | StrOutputParser()

# Chain with History
search_travel_chain_with_history = RunnableWithMessageHistory(
    search_travel_chain,
    lambda session_id: history,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="his",
)
other_tool__chain_chain_with_history = RunnableWithMessageHistory(
    other_tool_chain,
    lambda session_id: history,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="his",
)

if history.messages == []:
    history.add_ai_message("How may I assist you today?")

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)


# Chat Input - User Prompt
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    config = {"configurable": {"session_id": "any"}}
    print(prompt)
    pred_type = predict_conversation(history.messages, new_message=prompt)
    if pred_type == '旅遊規劃工具':
        if streaming_on:
            # Chain - Stream
            placeholder = st.empty()
            full_response = ''
            for chunk in search_travel_chain_with_history.stream({"question": prompt}, config):
                full_response += chunk
                placeholder.chat_message("ai").write(full_response)
            placeholder.chat_message("ai").write(full_response)

        else:
            # Chain - Invoke
            response = search_travel_chain_with_history.invoke(
                {"question": prompt}, config)
            st.chat_message("ai").write(response)
    elif pred_type == '其他工具':
        if streaming_on:
            # Chain - Stream
            placeholder = st.empty()
            full_response = ''
            for chunk in other_tool__chain_chain_with_history.stream({"question": prompt}, config):
                full_response += chunk
                placeholder.chat_message("ai").write(full_response)
            placeholder.chat_message("ai").write(full_response)

        else:
            # Chain - Invoke
            response = other_tool__chain_chain_with_history.invoke(
                {"question": prompt}, config)
            st.chat_message("ai").write(response)
