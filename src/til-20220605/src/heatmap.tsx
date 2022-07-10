import React from 'react';
import styles from './heatmap.css';

export const Heatmap = (): JSX.Element => {
  const rows = 5;
  const cols = 30;
  const dataList = Array.from({ length: rows }, () => getRandomList(cols));

  return (
    <div>
      <Header count={cols} />
      {dataList.map((data, idx) => (
        <Row rowName={idx.toString()} idx={idx} data={data} />
      ))}
    </div>
  );
};

const Header = (props: {count: number}): JSX.Element => {
  const dateList = Array.from({ length: props.count }, (v, k) => k);

  return (
    <div className={styles.header}>
      <div className={styles.leadingHeader}>Heat</div>
      {dateList.map((day) => {
        return <div className={styles.headerItem}>{day}</div>;
      })}
    </div>
  );
};

const Row = (props: { rowName: string; idx: number, data: number[] }): JSX.Element => {
  const red = 125;
  const green = 190;
  const blue = 130;

  return (
    <div className={styles.row}>
      <div className={styles.leadingRow}>{props.rowName}</div>
      {props.data.map((cellValue) => {
        const opacity = cellValue * 0.2 + 0.5;
        const background = `rgb(${red}, ${green}, ${blue}, ${opacity})`;

        return (
          <div className={styles.cell} style={{ background: background }}>
            {cellValue}
          </div>
        );
      })}
    </div>
  );
};

function getRandomList(count: number): number[] {
  return Array.from({ length: count }, () => getRandomInt());
}

function getRandomInt(): number {
  const maxValue = 10.0;
  return Math.floor(Math.random() * Math.floor(maxValue));
}
