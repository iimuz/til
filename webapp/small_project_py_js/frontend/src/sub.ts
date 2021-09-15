export const hello = (message: string): void => {
  log(message);
};

function log(message: string): void {
  // document.body.innerHTML = `${message}`;

  console.log(`output ${message}`);
}