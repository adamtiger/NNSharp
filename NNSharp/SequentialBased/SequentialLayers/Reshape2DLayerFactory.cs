using NNSharp.KernelDescriptors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Reshape2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is Reshape2D)
            {
                Reshape2D reshape = descriptor as Reshape2D;

                ILayer layer = new Reshape2DLayer(reshape.Height, reshape.Width, reshape.Channel, reshape.Batch);

                return layer;
            }

            return null;
        }
    }
}
