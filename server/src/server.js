const express = require('express');

const app = express();

app.get('/', (req, res) => {
  res.send('Hello World');
});

const port = 5001;

app.listen(port, () =>{
  console.log(`서버가 실행됩니다. ${port}`);
});