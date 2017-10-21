using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.IO;
using NNSharp.DataTypes;
using NNSharp.Models;
using static NNSharp.DataTypes.Data2D;
using UnitTests.Properties;

namespace UnitTests
{
    [TestClass]
    public class TestDropout
    {
        [TestMethod]
        public void Test_Dropout_KerasModel()
        {
            string pathModel = Resources.TestsFolder + "test_dropout_model.json";
            string pathInput = Resources.TestsFolder + "test_dropout_input.json";
            string pathOutput = Resources.TestsFolder + "test_dropout_output.json";

            var reader = new ReaderKerasModel(pathModel);
            SequentialModel model = reader.GetSequentialExecutor();

            // Initialize data.
            Data2D data = Utils.ReadDataFromFile(pathInput); 

            // Load expected output and calculate the actual output.
            Data2D expected = Utils.ReadDataFromFile(pathOutput);
            Data2D output = model.ExecuteNetwork(data) as Data2D;

            // Checking sizes
            Utils.CheckDimensions(output, expected);

            // Checking calculation
            double accuracy = 0.00001;
            Utils.CheckResults(output, expected, accuracy);
        }
    }
}