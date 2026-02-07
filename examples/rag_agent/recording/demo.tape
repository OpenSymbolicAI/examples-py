# RAG Agent Demo (Behaviour Mode)
# Generates: examples/rag_agent/assets/demo.gif

Output examples/rag_agent/assets/demo.gif

Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 14
Set Width 1800
Set Height 1100
Set Padding 20
Set LetterSpacing 0
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms

Type "cd examples/rag_agent && uv run python -m rag_agent.main"
Enter
Hide
Sleep 11s
Show

# Simple QA
Type "What is machine learning?"
Enter
Sleep 5s

# Summarization
Type "Summarize the causes of recent global temperature rise."
Enter
Sleep 5s

# Comparison
Type "Compare PostgreSQL and MongoDB for data storage"
Enter
Sleep 10s

# Multi-hop
Type "Who founded Tesla and what other companies did they start?"
Enter
Sleep 10s

Type "quit"
Enter
Sleep 500ms
