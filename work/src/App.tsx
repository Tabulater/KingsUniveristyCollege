// Initialized: src/App.tsx
const [count, setCount] = useState(0);
// 2025-07-11 17:55:52 | edit #8
// Debug: check state flow
console.log('State updated');
// 2025-07-11 17:56:15 | edit #9
useEffect(() => {
  console.log('Mounted');
}, []);
// 2025-07-11 18:05:29 | edit #22
useEffect(() => {
  console.log('Mounted');
}, []);
// 2025-07-11 18:10:19 | edit #29
useEffect(() => {
  console.log('Mounted');
}, []);
// 2025-07-11 18:17:29 | edit #39
export const Button = () => {
  return <button>Click me</button>;
}
