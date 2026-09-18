from index_documents import insert_documents
from search import search

# 原始文档
#   ↓
# 文本分块 Chunking
#   ↓
# Embedding  BAAI/bge-small-zh-v1.5 模型转成向量
#   ↓
# 写入 Chroma
#   ↓
# 用户问题转成向量
#   ↓
# 相似度搜索
#   ↓
# 返回相关文本
#   ↓
# 交给大模型生成答案


def search_and_print_results(q: str) -> None:
    query = "推荐给罗家馨什么好" 
    query = q
    results = search(query)
    print(f"\n查询：{query}")
    print("-" * 50)

    for index, (document, distance) in enumerate(
        zip(results["documents"][0], results["distances"][0]),
        start=1,
    ):
        print(f"第 {index} 名（相似度 {1 - distance:.4f}）：")
        print(f"  {document}\n")
        
        
def main() -> None:
    # documents = [
    #     '杨涛滔 搜索了六味地黄丸', #[2,3]
    #     '罗家馨 跟杨涛滔的地理位置高度重叠',#[2,4]
    #     '我高中都在玩游戏',#[1,-3]
    # ]
    # try:
    #     chunk_count = insert_documents(documents)
    #     print(f"已索引 {chunk_count} 个文本块")
    # except Exception as error:
    #     print(f"索引文档时出错: {error}")
    #     return
    
    
    search_and_print_results("推荐给罗家馨什么好")






if __name__ == "__main__":
    main()
