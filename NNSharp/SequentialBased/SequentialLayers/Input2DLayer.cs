using NNSharp.SequentialBased.SequentialLayers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;

namespace NNSharp.SequentialBased.SequentialLayers
{
    /**
     * Stores the sizes of the input data
     * The data containes zeros as default values 
     */

    public class Input2DLayer : ILayer
    {

        public Input2DLayer()
        {
            zerosInput = null;
        }

        public void Execute()
        {
             // nothing to do
        }

        public IData GetOutput()
        {
            return zerosInput;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("Input2DLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("Input2DLayer: input is not Data2D.");

            zerosInput = input as Data2D;
        }

        public void SetWeights(IData weights)
        {
            // No weights.
        }

        private Data2D zerosInput;
    }
}
