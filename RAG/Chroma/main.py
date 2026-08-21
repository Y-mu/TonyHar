from index_documents import index_documents
from search import search


def main() -> None:
    documents = [
        '杨涛滔 搜索了六味地黄丸', #[2,3]
        '罗家馨 跟杨涛滔的地理位置高度重叠',#[2,4]
        '我高中都在玩游戏',#[1,-3]
    ]
    metadatas = [
        {"source": "tutorial", "document_number": index}
        for index in range(len(documents))
    ]

    chunk_count = index_documents(documents, metadatas)
    print(f"已索引 {chunk_count} 个文本块")

    query = "推荐给罗家馨什么好" 
    results = search(query)
    print(f"\n查询：{query}")
    print("-" * 50)

    for index, (document, distance) in enumerate(
        zip(results["documents"][0], results["distances"][0]),
        start=1,
    ):
        print(f"第 {index} 名（相似度 {1 - distance:.4f}）：")
        print(f"  {document}\n")


if __name__ == "__main__":
    main()
