import { useState } from 'react';
import logo from './logo.svg';
import './App.css';

function App() {
  const [description, setDescription] = useState("Learn React");
  const changeDescription = async () => {
    setMessage(setDescription)
  }

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <button onClick={changeDescription}>{description}</button>
      </header>
    </div>
  );
}

function setMessage(setFunc) {
  fetch("/hello")
    .then(async response => {
      const data = await response.json();
      if (!response.ok) {
        console.log('test')
        const error = (data && data.message) || response.statusText;
        return Promise.reject(error);
      }

      setFunc(data['message'])
    })
    .catch(error => {
      console.error('There wa an error: ', error)
    });
}

export default App;
