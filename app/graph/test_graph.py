from app.graph.graph import graph


result = graph.invoke(
    {
        # This is the only information we need to provide initially.
        # The remaining state fields will be produced by graph nodes.
        "question": "Who was talking about the project yesterday?",
    }
)


print("\n========== FINAL ANSWER ==========")
print(result["answer"])