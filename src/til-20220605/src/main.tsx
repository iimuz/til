import React from 'react';
import ReactDOM from 'react-dom';

import { Heatmap } from './heatmap';

const App = (): JSX.Element => {
  return (
    <div>
      <h1>Heatmap Table</h1>
      <Heatmap />
    </div>
  );
};

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);
