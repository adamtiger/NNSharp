using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Input2DLayerFactory : ILayerFactory
    {
        public ILayer CreateProduct(IKernelDescriptor descriptor)
        {
            if (descriptor is Input2D)
            {
                Input2D inputDescriptor = descriptor as Input2D;

                Data2D data = new Data2D(inputDescriptor.Height, inputDescriptor.Width,
                                         inputDescriptor.Channels, inputDescriptor.Batch);

                data.ToZeros();
                Input2DLayer layer = new Input2DLayer();
                layer.SetInput(data);

                return layer;
            }

            return null;
        }
    }
}
