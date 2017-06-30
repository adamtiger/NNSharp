using NNSharp.DataTypes;
using NNSharp.IO;
using NNSharp.KernelDescriptors;
using NNSharp.Models;
using NNSharp.SequentialBased.SequentialExecutors;
using NNSharp.SequentialBased.SequentialLayers;
using System;
using System.Collections.Generic;
using System.Text;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // Keras speed with the same: 60 ms.
            /*ReaderKerasModel reader = new ReaderKerasModel("test_cnn_model.json");
            SequentialModel model = reader.GetSequentialExecutor();

            Console.WriteLine((model.GetSummary() as SequentialModelData).GetStringRepresentation());

            Console.ReadKey();
            int[] idx = { 1,2,3};

            Console.WriteLine(idx[1]);
            Console.ReadKey();*/

            /*Conv2DLayer layer = new Conv2DLayer(0, 0, 1, 1);

            Data2D input = new Data2D(6, 5, 3, 1);

            input[0, 0, 0, 0] = 1; input[0, 1, 0, 0] =  2; input[0, 2, 0, 0] =  2; input[0, 3, 0, 0] =  1; input[0, 4, 0, 0] = 4;
            input[1, 0, 0, 0] = 3; input[1, 1, 0, 0] =  1; input[1, 2, 0, 0] =  0; input[1, 3, 0, 0] =  2; input[1, 4, 0, 0] = 1;
            input[2, 0, 0, 0] = 0; input[2, 1, 0, 0] =  2; input[2, 2, 0, 0] =  2; input[2, 3, 0, 0] =  5; input[2, 4, 0, 0] = 2;
            input[3, 0, 0, 0] = 6; input[3, 1, 0, 0] = -2; input[3, 2, 0, 0] = -1; input[3, 3, 0, 0] =  3; input[3, 4, 0, 0] = 1;
            input[4, 0, 0, 0] = 2; input[4, 1, 0, 0] =  1; input[4, 2, 0, 0] =  2; input[4, 3, 0, 0] =  4; input[4, 4, 0, 0] = 0;
            input[5, 0, 0, 0] = 5; input[5, 1, 0, 0] = -3; input[5, 2, 0, 0] = -1; input[5, 3, 0, 0] = -4; input[5, 4, 0, 0] = 0;

            input[0, 0, 1, 0] = 2; input[0, 1, 1, 0] =  0; input[0, 2, 1, 0] =  2; input[0, 3, 1, 0] = -1; input[0, 4, 1, 0] = 3;
            input[1, 0, 1, 0] = 2; input[1, 1, 1, 0] =  5; input[1, 2, 1, 0] = -1; input[1, 3, 1, 0] =  3; input[1, 4, 1, 0] = 5;
            input[2, 0, 1, 0] = 1; input[2, 1, 1, 0] =  1; input[2, 2, 1, 0] =  1; input[2, 3, 1, 0] =  0; input[2, 4, 1, 0] = 1;
            input[3, 0, 1, 0] =-3; input[3, 1, 1, 0] =  2; input[3, 2, 1, 0] = -1; input[3, 3, 1, 0] =  4; input[3, 4, 1, 0] = 1;
            input[4, 0, 1, 0] = 2; input[4, 1, 1, 0] =  1; input[4, 2, 1, 0] =  2; input[4, 3, 1, 0] =  2; input[4, 4, 1, 0] = 1;
            input[5, 0, 1, 0] = 0; input[5, 1, 1, 0] = -3; input[5, 2, 1, 0] =  1; input[5, 3, 1, 0] = -2; input[5, 4, 1, 0] =-1;

            input[0, 0, 2, 0] = 4; input[0, 1, 2, 0] = 5; input[0, 2, 2, 0] = 0; input[0, 3, 2, 0] =-1; input[0, 4, 2, 0] =-3;
            input[1, 0, 2, 0] = 2; input[1, 1, 2, 0] = 3; input[1, 2, 2, 0] = 1; input[1, 3, 2, 0] = 6; input[1, 4, 2, 0] = 0;
            input[2, 0, 2, 0] = 0; input[2, 1, 2, 0] =-4; input[2, 2, 2, 0] =-3; input[2, 3, 2, 0] =-2; input[2, 4, 2, 0] =-4;
            input[3, 0, 2, 0] = 4; input[3, 1, 2, 0] = 2; input[3, 2, 2, 0] = 1; input[3, 3, 2, 0] = 0; input[3, 4, 2, 0] = 4;
            input[4, 0, 2, 0] = 3; input[4, 1, 2, 0] = 3; input[4, 2, 2, 0] = 0; input[4, 3, 2, 0] = 1; input[4, 4, 2, 0] = 1;
            input[5, 0, 2, 0] =-2; input[5, 1, 2, 0] = 1; input[5, 2, 2, 0] = 1; input[5, 3, 2, 0] = 0; input[5, 4, 2, 0] = 5;

            Data2D kernel = new Data2D(3, 3, 3, 1);

            kernel[0, 0, 0, 0] = 1; kernel[0, 1, 0, 0] = 1; kernel[0, 2, 0, 0] = 0;
            kernel[1, 0, 0, 0] = 2; kernel[1, 1, 0, 0] = 0; kernel[1, 2, 0, 0] = 0;
            kernel[2, 0, 0, 0] = 1; kernel[2, 1, 0, 0] = 2; kernel[2, 2, 0, 0] = 1;

            kernel[0, 0, 1, 0] = 3; kernel[0, 1, 1, 0] = 1; kernel[0, 2, 1, 0] =-1;
            kernel[1, 0, 1, 0] = 2; kernel[1, 1, 1, 0] =-1; kernel[1, 2, 1, 0] =-2;
            kernel[2, 0, 1, 0] = 0; kernel[2, 1, 1, 0] = 1; kernel[2, 2, 1, 0] = 2;

            kernel[0, 0, 2, 0] = 0; kernel[0, 1, 2, 0] = 1; kernel[0, 2, 2, 0] = 1;
            kernel[1, 0, 2, 0] =-1; kernel[1, 1, 2, 0] = 2; kernel[1, 2, 2, 0] = 1;
            kernel[2, 0, 2, 0] = 3; kernel[2, 1, 2, 0] = 0; kernel[2, 2, 2, 0] = 1;

            layer.SetWeights(kernel);
            layer.SetInput(input);
            layer.Execute();

            Data2D output = layer.GetOutput() as Data2D;*/

            Dimension a1 = new Dimension(1, 3, 5, 2);
            Dimension a2 = new Dimension(1, 3, 5, 2);

            if (a1.Equals(a2))
                Console.WriteLine("Helyes.");
            else
                Console.WriteLine("Hibás.");

            Console.ReadKey();

        }
    }
}
