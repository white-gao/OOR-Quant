# Phase 1: Generator Agent
# 这里的 "Agent" 其实就是一个函数
def run_generator_agent(target_item_metadata, llm_client):
    prompt = f"""
    目标物品信息: {target_item_metadata}
    任务: 请生成一个用户寻找该物品的复杂请求，不要直接提名字。
    要求: 需要经过3步推理。
    """
    # 调用通用大模型 (Teacher)
    question = llm_client.generate(prompt)
    return question

# Phase 2: Solver Agent
# 这里的 "Agent" 其实就是 OpenOneRec 的 inference
def run_search_agent(question, openonerec_model):
    # OpenOneRec 接收问题，输出 Token
    # SAGE 论文中提到 Search Agent 输出推理轨迹和答案 [cite: 93]
    reasoning_trace, predicted_token_id = openonerec_model.inference(question)
    return predicted_token_id, reasoning_trace

# Phase 3: Pipeline
def sage_pipeline(target_item):
    # 1. 初始出题
    question = run_generator_agent(target_item.metadata)
    
    # 循环优化 (SAGE 论文中的 Feedback Loop) 
    for round in range(MAX_ROUNDS):
        # 2. 做题
        pred_token, trace = run_search_agent(question)
        
        # 3. 翻译 (我建议添加的步骤)
        pred_item_text = item_id_to_text(pred_token)
        
        # 4. 判卷 (让 Teacher 看看对不对)
        feedback_prompt = f"""
        原定目标: {target_item.metadata}
        模型回答: {pred_item_text}
        问题: {question}
        
        请判断:
        1. 模型回答是否正确？
        2. 这个问题对模型来说是否太简单（一步就猜到了）？
        
        如果太简单或答错了，请给出修改意见并重写问题。
        """
        evaluation = llm_client.generate(feedback_prompt)
        
        if "合格" in evaluation:
            return question, target_item # 数据生成成功！
        else:
            question = extract_new_question(evaluation) # 获取改进后的问题，进入下一轮
            
    return None # 尝试多次失败，丢弃这条数据