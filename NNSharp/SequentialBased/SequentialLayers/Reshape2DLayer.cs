using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;
using static NNSharp.DataTypes.Data2D;
using static NNSharp.DataTypes.SequentialModelData;

namespace NNSharp.SequentialBased.SequentialLayers
{
    [Serializable()]
    public class Reshape2DLayer : Reshape2DKernel, ILayer
    {
        public Reshape2DLayer(int height, int width, int channel, int batch)
        {
            this.height = height;
            this.width = width;
            this.channel = channel;
            this.batch = batch;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("Reshape2DLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("Reshape2DLayer: input is not Data2D.");

            this.input = input as Data2D;
            Dimension dimI = this.input.GetDimension();

            int numInputElements = dimI.b * dimI.c * dimI.h * dimI.w;
            int numOutputElements = batch * channel * height * width; 

            if (numInputElements != numOutputElements)
                throw new Exception("The number of elements in input and output are different." +
                   " Now: numInputElements: " + numInputElements + " != numOutputElements: " + numOutputElements);

            output = new Data2D(height, width, channel, batch);
        }

        public void SetWeights(IData weights)
        {
            // No weights.
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


        private int height;
        private int width;
        private int channel;
        private int batch;
    }
}
