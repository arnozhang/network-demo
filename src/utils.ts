import { shuffle } from 'lodash';

/**
 * 打乱数组顺序
 */
export function shuffleList<T>(list: T[]): T[] {
  return shuffle(list);
}

/**
 * MSE Loss 计算
 */
export function mseLoss(y: number, yPredict: number): number {
  const offset = y - yPredict;
  return (offset * offset) / 2;
}

/**
 * 正态分布随机数
 */
export function initializeGaussianRandom(count: number): number[] {
  return new Array(count).fill(0).map(() => gaussianRandom());
}

/**
 * 生成标准正态分布随机数（Box-Muller 变换）
 */
export function gaussianRandom(): number {
  let u = 0;
  let v = 0;

  while (u === 0) {
    u = Math.random();
  }

  while (v === 0) {
    v = Math.random();
  }

  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * 标准化
 */
export function standardizeArray(array: number[]) {
  const mean = array.reduce((acc, val) => acc + val, 0) / array.length;
  const deviation = Math.sqrt(
    array.map(val => (val - mean) ** 2).reduce((acc, val) => acc + val, 0) /
    array.length,
  );

  const standardize = (input: number) => {
    return (input - mean) / deviation;
  };

  return {
    list: array.map(standardize),
    standardize,
    mean,
    deviation,
  };
}
