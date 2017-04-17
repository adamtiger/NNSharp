using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Dense2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is Dense2D)
            {
                Dense2D dens = descriptor as Dense2D;

                Data2D weights = new Data2D(1, 1, dens.Units, 1);

                ILayer layer = new Dense2DLayer(weights);

                return layer;
            }

            return null;
        }
    }
}
