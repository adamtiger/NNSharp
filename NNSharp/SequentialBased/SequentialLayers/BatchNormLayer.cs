using NNSharp.Kernels.CPUKernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using static NNSharp.DataTypes.SequentialModelData;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class BatchNormLayer : BatchNormKernel, ILayer
    {

        public BatchNormLayer(double epsilon)
        {
            this.epsilon = epsilon;
            this.gamma = new List<double>();
            this.beta = new List<double>();
            this.bias = new List<double>();
            this.variance = new List<double>();
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("BatchNormLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("BatchNormLayer: input is not Data2D.");

            this.input = input as Data2D;

            Dimension dimI = this.input.GetDimension();
            int kGamma = this.gamma.Count;
            int kBeta = this.beta.Count;
            int kBias = this.bias.Count;
            int kVariance = this.variance.Count;

            if (dimI.c != kBias || dimI.c != kGamma || dimI.c != kBeta || dimI.c != kVariance)
                throw new Exception("Number of parameters is not equal with number of features (channels).");

            int outputH = dimI.h;
            int outputW = dimI.w;
            int outputC = dimI.c;
            int outputB = dimI.b;

            output = new Data2D(outputH, outputW, outputC, outputB);
        }

        public void SetWeights(IData parameters)
        {
            if (parameters == null)
                throw new Exception("BatchNormLayer: parameters is null.");
            else if (!(parameters is Data2D))
                throw new Exception("BatchNormLayer: parameters is not Data2D.");

            Data2D pms = parameters as Data2D;

            if (pms.GetDimension().h != 1 || pms.GetDimension().w != 1)
                throw new Exception("BatchNormLayer: parameters' height and width should be 1.");

            for (int feature = 0; feature < pms.GetDimension().c; ++feature)
            {
                gamma.Add(pms[0, 0, feature, 0]);
                beta.Add(pms[0, 0, feature, 1]);
                bias.Add(pms[0, 0, feature, 2]);
                variance.Add(pms[0, 0, feature, 3]);
            }
        }

        public LayerData GetLayerSummary()
        {
            Dimension dimI = input.GetDimension();
            Dimension dimO = output.GetDimension();
            return new LayerData(
                this.ToString(),
                dimI.h, dimI.w, 1, dimI.c, dimI.b,
                dimO.h, dimO.w, 1, dimO.c, dimO.b);
        }
    }
}
