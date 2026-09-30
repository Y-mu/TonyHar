"""TonyHar 命令行入口。"""

import asyncio

from tonyhar.bootstrap import build_chat_service


async def main() -> None:
    chat_service = build_chat_service()
    session_id = "cli"

    try:
        while True:
            user_input = (await asyncio.to_thread(input, "你: ")).strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            result = await chat_service.invoke(session_id, user_input)
            if result.success:
                print("Agent:", result.answer)
            else:
                print("Agent error:", result.error_message or result.answer)
    finally:
        await chat_service.aclose()


if __name__ == "__main__":
    asyncio.run(main())
