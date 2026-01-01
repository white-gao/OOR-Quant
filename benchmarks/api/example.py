"""
LLM API 使用示例
展示各种调用方式和使用场景
"""

# ============================================================================
# 示例1: 使用配置文件（最简单）
# ============================================================================
def example1_use_config():
    """从配置文件加载并使用"""
    from api import get_client_from_config

    print("=" * 60)
    print("示例1: 使用配置文件")
    print("=" * 60)

    # 从配置文件创建客户端
    client = get_client_from_config("gemini")

    # 生成文本
    response = client.generate("用一句话解释什么是AI")
    print(f"回答: {response}\n")


# ============================================================================
# 示例2: 直接传参
# ============================================================================
def example2_direct_params():
    """直接传递配置参数"""
    from api import get_client

    print("=" * 60)
    print("示例2: 直接传递参数")
    print("=" * 60)

    # Gemini
    gemini_client = get_client(
        "gemini",
        project="your-project",
        location="us-central1",
        model_name="gemini-2.5-pro",
        credentials_path="path/to/credentials.json"
    )

    # DeepSeek
    deepseek_client = get_client(
        "deepseek",
        api_key="your-api-key",
        appid="your-appid",
        base_url="https://qianfan.baidubce.com/v2"
    )

    # 使用
    response = gemini_client.generate("你好")
    print(f"Gemini: {response}\n")


# ============================================================================
# 示例3: 批量生成（并发）
# ============================================================================
def example3_batch_generate():
    """批量生成文本，支持并发"""
    from api import get_client_from_config

    print("=" * 60)
    print("示例3: 批量生成（并发）")
    print("=" * 60)

    prompts = [
        "什么是机器学习？",
        "解释一下深度学习",
        "神经网络的原理",
        "什么是自然语言处理？",
        "计算机视觉的应用"
    ]

    # 使用客户端实例的batch_generate方法（推荐）
    client = get_client_from_config("gemini")
    results = client.batch_generate(
        prompts=prompts,
        max_workers=3,  # 3个并发线程
        show_progress=True  # 显示进度条
    )

    # 处理结果
    for i, item in enumerate(results, 1):
        print(f"\n问题{i}: {item['prompt']}")
        if item['success']:
            print(f"回答: {item['result'][:100]}...")
        else:
            print(f"错误: {item['error']}")


# ============================================================================
# 示例4: 控制生成参数
# ============================================================================
def example4_custom_params():
    """自定义生成参数"""
    from api import get_client_from_config

    print("=" * 60)
    print("示例4: 自定义生成参数")
    print("=" * 60)

    client = get_client_from_config("deepseek")

    # 创造性生成（高temperature）
    creative = client.generate(
        "写一首关于春天的诗",
        temperature=0.9,
        max_tokens=200
    )
    print(f"创造性输出:\n{creative}\n")

    # 精确生成（低temperature）
    precise = client.generate(
        "1+1等于几？",
        temperature=0.1,
        max_tokens=50
    )
    print(f"精确输出:\n{precise}\n")


# ============================================================================
# 示例5: 错误处理
# ============================================================================
def example5_error_handling():
    """展示错误处理"""
    from api import get_client_from_config

    print("=" * 60)
    print("示例5: 错误处理")
    print("=" * 60)

    try:
        client = get_client_from_config("gemini")

        # 正常调用
        response = client.generate("你好")
        print(f"成功: {response}")

        # 空提示词（会抛出ValueError）
        response = client.generate("")

    except ValueError as e:
        print(f"参数错误: {e}")
    except Exception as e:
        print(f"API调用失败: {e}")


# ============================================================================
# 示例6: 切换模型
# ============================================================================
def example6_switch_models():
    """在不同模型间切换"""
    from api import get_client_from_config

    print("=" * 60)
    print("示例6: 切换模型")
    print("=" * 60)

    question = "什么是量子计算？"

    for model_name in ["gemini", "deepseek"]:
        try:
            client = get_client_from_config(model_name)
            response = client.generate(question)
            print(f"\n{model_name.upper()}的回答:")
            print(response[:150] + "...")
        except Exception as e:
            print(f"\n{model_name}调用失败: {e}")


# ============================================================================
# 示例7: 实际应用 - 用户画像生成
# ============================================================================
def example7_user_portrait():
    """实际应用：根据用户行为生成用户画像"""
    from api import get_client_from_config

    print("=" * 60)
    print("示例7: 用户画像生成")
    print("=" * 60)

    # 用户行为数据
    user_behavior = """
    用户最近观看的视频：
    1. 机器学习入门教程
    2. Python编程技巧
    3. 深度学习实战项目
    4. 数据分析案例
    5. AI领域最新动态
    """

    prompt = f"""根据以下用户行为数据，生成一份简洁的用户画像：

{user_behavior}

要求：
1. 总结用户的兴趣领域
2. 推测用户的技能水平
3. 给出3-5个精准的标签
"""

    client = get_client_from_config("gemini")
    portrait = client.generate(prompt, temperature=0.5)

    print("用户画像:")
    print(portrait)


# ============================================================================
# 示例8: 使用直接导入的类
# ============================================================================
def example8_direct_import():
    """直接导入客户端类"""
    from api import GeminiClient, DeepSeekClient

    print("=" * 60)
    print("示例8: 直接导入客户端类")
    print("=" * 60)

    # 直接实例化
    gemini = GeminiClient(
        project="your-project",
        location="us-central1"
    )

    deepseek = DeepSeekClient(
        api_key="your-key",
        appid="your-appid"
    )

    print("客户端创建成功")
    print(f"Gemini客户端: {gemini}")
    print(f"DeepSeek客户端: {deepseek}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """运行所有示例"""
    examples = [
        ("使用配置文件", example1_use_config),
        ("直接传参", example2_direct_params),
        ("批量生成", example3_batch_generate),
        ("自定义参数", example4_custom_params),
        ("错误处理", example5_error_handling),
        ("切换模型", example6_switch_models),
        ("用户画像生成", example7_user_portrait),
        ("直接导入类", example8_direct_import),
    ]

    print("\n" + "=" * 60)
    print("LLM API 使用示例集")
    print("=" * 60)
    print("\n可用示例：")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")

    print("\n提示：运行前请确保已配置 api/config/llm_config.json")
    print("\n" + "=" * 60 + "\n")

    # 取消注释下面的行来运行特定示例
    # example1_use_config()
    # example2_direct_params()
    # example3_batch_generate()
    # example4_custom_params()
    # example5_error_handling()
    # example6_switch_models()
    # example7_user_portrait()
    # example8_direct_import()


if __name__ == "__main__":
    main()
