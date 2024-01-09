import { plot } from 'nodeplotlib';
import { SimpleNetwork } from '@/network';
import { startTrain } from '@/train';
import { standardizeArray } from '@/utils';

function start() {
  // 1. 构造数据集: f(x) = (x * 2 + 3.5) / 1000
  const trainingData = new Array(1000).fill(0).map((_, index) => {
    const input = Math.random() * index;
    return {
      input,
      target: (input * 2 + 3.5) / 1000,
    };
  });

  // 2. 数据集输入标准化
  const { standardize, list } = standardizeArray(
    trainingData.map(item => item.input),
  );
  trainingData.forEach((item, index) => (item.input = list[index]));

  // 3. 创建一个简单网络: 隐藏层有 3 个神经元
  const network = new SimpleNetwork(3);

  // 4. 训练网络
  const lossData = startTrain(network, trainingData);

  // 5. 打印网络结构
  console.log('train finished.\n');
  network.printNetwork();

  // 6. 展示 Loss
  plot([
    {
      ...lossData,
      xaxis: 'epoch',
      yaxis: 'Loss',
      type: 'scatter',
    },
  ]);

  // 7. 测试网络
  const testInput = 4096;

  // 7.1. 将输入标准化
  const testInputValue = standardize(testInput);

  // 7.2. 模型推理
  const predictedOutput = network.predict([testInputValue]);

  console.log(`\nPredicted output for input ${testInput}: ${predictedOutput}`);
}

start();
