using NNSharp.DataTypes;
using NNSharp.IO;
using NNSharp.Models;
using System;
using System.Text;

namespace NNSharp
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // Keras speed with the same: 60 ms.
            /*ReaderKerasModel reader = new ReaderKerasModel("test_cnn_model.json");
            SequentialModel model = reader.GetSequentialExecutor();

            Console.WriteLine(model.GetSummary().GetStringRepresentation());

            Console.ReadKey();*/
            int[] idx = { 1,2,3};

            Console.WriteLine(idx[1]);
            Console.ReadKey();

        }
    }
}
