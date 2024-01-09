import { Neuron } from '@/neuron';
import { initializeGaussianRandom } from '@/utils';

interface IGradientInfo {
  bias: number;
  weights: number[];
}

interface INetworkGradient {
  outputGradient: IGradientInfo;
  hiddenLayerGradient: IGradientInfo[];
}

/**
 * 只有一个隐藏层的简单神经网络定义
 */
export class SimpleNetwork {
  private readonly hiddenLayer: Neuron[];
  private readonly outputNeuron: Neuron;

  constructor(hiddenLayerSize: number, learningRate: number = 0.01) {
    // 正态分布小数随机初始化网络参数
    const weights = initializeGaussianRandom(hiddenLayerSize * 2);

    this.hiddenLayer = new Array(hiddenLayerSize).fill(null).map(
      (_, i) =>
        new Neuron({
          weights: [ weights[i] ],
          learningRate,
        }),
    );

    this.outputNeuron = new Neuron({
      weights: weights.slice(hiddenLayerSize),
      learningRate,
    });
  }

  predict(input: number[]): number {
    const hiddenLayerOutputs = this.hiddenLayer.map(neuron =>
      neuron.feedforward(input),
    );

    return this.outputNeuron.feedforward(hiddenLayerOutputs);
  }

  train(input: number, target: number): number {
    // 前向传播
    const output = this.predict([ input ]);

    // 计算梯度
    const gradient = this.evaluateGradient(output, target);

    // 反向传播梯度
    this.backward(gradient);

    return output;
  }

  /**
   * 计算梯度
   */
  private evaluateGradient(output: number, target: number): INetworkGradient {
    const outputBias = output - target;

    return {
      outputGradient: {
        bias: outputBias,
        weights: this.outputNeuron.inputs.map(input => {
          return outputBias * input;
        }),
      },
      hiddenLayerGradient: this.hiddenLayer.map((neuron, i) => {
        const hiddenLayerBias =
          (output - target) * this.outputNeuron.weights[i];

        return {
          bias: hiddenLayerBias,
          weights: [ hiddenLayerBias * neuron.inputs[0] ],
        };
      }),
    };
  }

  /**
   * 反向传播更新参数
   */
  private backward(gradient: INetworkGradient) {
    const { outputGradient, hiddenLayerGradient } = gradient;

    // 更新输出层神经元的权重
    this.outputNeuron.updateWeights(
      outputGradient.weights,
      outputGradient.bias,
    );

    // 更新隐藏层神经元的权重
    this.hiddenLayer.forEach((neuron, i) => {
      const { weights, bias } = hiddenLayerGradient[i];
      neuron.updateWeights(weights, bias);
    });
  }

  /**
   * 打印网络结构
   */
  printNetwork() {
    console.log('-- outputLayer: ');
    this.outputNeuron.print();

    console.log(`\n-- hiddenLayer: size = ${this.hiddenLayer.length}`);
    this.hiddenLayer.forEach((item, index) => {
      console.log(`\n-- layer item ${index}:`);
      item.print();
    });
  }
}
