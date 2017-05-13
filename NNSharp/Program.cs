using NNSharp.DataTypes;
using NNSharp.IO;
using NNSharp.Models;
using System;


namespace NNSharp
{
    public class Program
    {
        public static void Main(string[] args)
        {

            TestAvgPool2D1();

            Console.WriteLine("Finished.");
        }

        private static void Common(string fpath)
        {
            var reader = new ReaderKerasModel(fpath);

            SequentialModel model1 = reader.GetSequentialExecutor();

            PersistSequentialModel.SerializeModel(model1, "test.bin");

            SequentialModel model = PersistSequentialModel.DeserializeModel("test.bin");

            Data2D inp = new Data2D(4, 5, 2, 1);

            inp[0, 0, 0, 0] = 0;
            inp[0, 0, 1, 0] = 1;
            inp[0, 1, 0, 0] = 2;
            inp[0, 1, 1, 0] = 1;
            inp[0, 2, 0, 0] = 0;
            inp[0, 2, 1, 0] = 0;
            inp[0, 3, 0, 0] = 2;
            inp[0, 3, 1, 0] = 1;
            inp[0, 4, 0, 0] = 2;
            inp[0, 4, 1, 0] = 1;


            inp[1, 0, 0, 0] = 0;
            inp[1, 0, 1, 0] = -1;
            inp[1, 1, 0, 0] = 1;
            inp[1, 1, 1, 0] = -2;
            inp[1, 2, 0, 0] = 3;
            inp[1, 2, 1, 0] = 1;
            inp[1, 3, 0, 0] = 2;
            inp[1, 3, 1, 0] = 0;
            inp[1, 4, 0, 0] = 2;
            inp[1, 4, 1, 0] = -3;


            inp[2, 0, 0, 0] = 1;
            inp[2, 0, 1, 0] = 2;
            inp[2, 1, 0, 0] = -2;
            inp[2, 1, 1, 0] = 0;
            inp[2, 2, 0, 0] = 3;
            inp[2, 2, 1, 0] = -3;
            inp[2, 3, 0, 0] = 2;
            inp[2, 3, 1, 0] = 1;
            inp[2, 4, 0, 0] = 2;
            inp[2, 4, 1, 0] = 0;


            inp[3, 0, 0, 0] = 1;
            inp[3, 0, 1, 0] = 2;
            inp[3, 1, 0, 0] = 0;
            inp[3, 1, 1, 0] = -2;
            inp[3, 2, 0, 0] = 3;
            inp[3, 2, 1, 0] = 1;
            inp[3, 3, 0, 0] = 2;
            inp[3, 3, 1, 0] = 3;
            inp[3, 4, 0, 0] = -3;
            inp[3, 4, 1, 0] = 1;

            IData ou = model.ExecuteNetwork(inp);
        }

        private static void TestConv2D1()
        {
            Common("tests/test_conv_2D_1_model.json");
        }

        private static void TestConv2D2()
        {
            Common("tests/test_conv_2D_2_model.json");
        }

        private static void TestAvgPool2D1()
        {
            Common("tests/test_avgpool_2D_1_model.json");
        }

        private static void TestAvgPool2D2()
        {
            Common("tests/test_avgpool_2D_2_model.json");
        }

        private static void TestMaxPool2D1()
        {
            Common("tests/test_maxpool_2D_1_model.json");
        }

        private static void TestMaxPool2D2()
        {
            Common("tests/test_maxpool_2D_2_model.json");
        }

        private static void TestFlatten()
        {
            Common("tests/test_flat_model.json");
        }

        private static void TestDense()
        {
            var reader = new ReaderKerasModel("tests/test_dense_model.json");

            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(1, 8, 1, 1);

            inp[0, 0, 0, 0] = 1;
            inp[0, 1, 0, 0] = 2;
            inp[0, 2, 0, 0] = -1;
            inp[0, 3, 0, 0] = 0;

            inp[0, 4, 0, 0] = 3;
            inp[0, 5, 0, 0] = 1;
            inp[0, 6, 0, 0] = 1;
            inp[0, 7, 0, 0] = 2;

            IData ou = model.ExecuteNetwork(inp);
        }


    }
}
