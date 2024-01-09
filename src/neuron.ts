/**
 * 神经元定义
 */
export class Neuron {
  /**
   * 权重
   */
  readonly weights: number[];

  /**
   * 偏置
   */
  bias: number;

  /**
   * 学习率
   */
  private readonly learningRate: number;

  private _inputs: number[];
  private _sum: number;
  private _output: number;

  constructor(config: {
    weights: number[];
    bias?: number;
    learningRate: number;
  }) {
    this.weights = config.weights;
    this.bias = config.bias || 0.01;
    this.learningRate = config.learningRate;
  }

  get inputs(): number[] {
    return this._inputs;
  }

  get sum(): number {
    return this._sum;
  }

  get output(): number {
    return this._output;
  }

  /**
   * 计算神经元的输出
   */
  feedforward(inputs: number[]): number {
    if (inputs.length !== this.weights.length) {
      // 确保输入值的数量与权重数量相等
      throw new Error(
        `invalid inputValues length: ${inputs.length} != ${this.weights.length}`,
      );
    }

    // 计算加权输入和偏置的和
    let sum = this.bias;
    this.weights.forEach((weight, i) => {
      sum += weight * inputs[i];
    });

    // 记录下本次运算值
    this._inputs = inputs;
    this._sum = sum;
    this._output = sum;

    return sum;
  }

  updateWeights(weights: number[], bias: number): void {
    // 更新每个权重
    this.weights.forEach((weight, i) => {
      this.weights[i] = weight - weights[i] * this.learningRate;
    });

    // 更新偏置
    this.bias -= bias * this.learningRate;
  }

  print() {
    console.log('-- weights: ', this.weights.join(', '));
    console.log('-- bias: ', this.bias);
  }
}
