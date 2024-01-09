import { SimpleNetwork } from '@/network';
import { mseLoss, shuffleList } from '@/utils';

interface ILossData {
  x: number[];
  y: number[];
}

export function startTrain(
  network: SimpleNetwork,
  trainingData: {
    input: number;
    target: number;
  }[],
): ILossData {
  const lossData = {
    x: [],
    y: [],
  };

  for (let epoch = 0; epoch < 20; ++epoch) {
    console.log(`-- epoch: ${epoch}`);

    // 1. 数据集乱序
    const shuffled = shuffleList(trainingData);

    let sumLoss = 0;
    for (const data of shuffled) {
      const output = network.train(data.input, data.target);

      // 2. 计算 Loss
      const loss = mseLoss(data.target, output);
      sumLoss += loss;
    }

    // 3. epoch 结束, 记录 Loss
    if (epoch % 2 === 0) {
      lossData.x.push(epoch);
      lossData.y.push(sumLoss / shuffled.length);
    }
  }

  return lossData;
}
