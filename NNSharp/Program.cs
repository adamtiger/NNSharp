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

            ReaderKerasModel reader = new ReaderKerasModel("test_elu_model.json");
            SequentialModel model = reader.GetSequentialExecutor();

            Console.WriteLine(model.GetSummary().GetStringRepresentation());

            Console.ReadKey();
        }

    }
}
