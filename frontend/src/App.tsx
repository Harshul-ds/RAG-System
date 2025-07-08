import { useState, type FormEvent, type ChangeEvent } from 'react';
import axios from 'axios';
import { cn } from "@/lib/utils";

interface Message {
  role: 'user' | 'bot';
  text: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post('/api/ask', { question: input });
      const botMessage: Message = { role: 'bot', text: response.data.answer };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Error fetching response:', error);
      const errorMessage: Message = { role: 'bot', text: 'Sorry, something went wrong.' };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white p-4">
      <header className="text-center mb-4">
        <h1 className="text-3xl font-bold">Agentic RAG</h1>
        <p className="text-gray-400">Your AI-powered research assistant</p>
      </header>

      <div className="flex-1 overflow-y-auto bg-gray-800 rounded-lg p-4 space-y-4">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={cn(
              "flex items-start gap-3",
              msg.role === 'user' ? 'justify-end' : 'justify-start'
            )}
          >
            <div
              className={cn(
                "p-3 rounded-lg max-w-lg",
                msg.role === 'user' ? 'bg-blue-600' : 'bg-gray-700'
              )}
            >
              <p className="whitespace-pre-wrap">{msg.text}</p>
            </div>
          </div>
        ))}
        {isLoading && (
            <div className="flex justify-start gap-3">
                <div className="p-3 rounded-lg bg-gray-700">
                    <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse [animation-delay:0.2s]"></div>
                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse [animation-delay:0.4s]"></div>
                    </div>
                </div>
            </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="mt-4 flex items-center">
        <input
          type="text"
          value={input}
          onChange={(e: ChangeEvent<HTMLInputElement>) => setInput(e.target.value)}
          placeholder="Ask a question..."
          className="flex-1 p-3 bg-gray-700 border border-gray-600 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        />
        <button
          type="submit"
          className="p-3 bg-blue-600 rounded-r-lg hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed"
          disabled={isLoading}
        >
          Send
        </button>
      </form>
    </div>
  );
}

export default App;
