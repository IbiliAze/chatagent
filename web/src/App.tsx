import ChatWidget from './ChatWidget';

export default function App() {
  return (
    <div className="page">
      <h1>ChatAgent UI Tester</h1>
      <p>Use the chat bubble in the bottom-right corner to talk to the API.</p>
      <ChatWidget />
    </div>
  );
}
