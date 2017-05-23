using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class RepeatVectorLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is RepeatVector)
            {
                RepeatVector repeat = descriptor as RepeatVector;

                ILayer layer = new RepeatVectorLayer(repeat.Num);

                return layer;
            }

            return null;
        }
    }
}
