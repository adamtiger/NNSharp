using NNSharp.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.IO
{
    public class PersistSequentialModel
    {
        public static void SerializeModel(SequentialModel model, string fileName)
        {
            Stream stream = File.Create(fileName);
            BinaryFormatter serializer = new BinaryFormatter();
            serializer.Serialize(stream, model);
            stream.Close();
        }

        public static SequentialModel DeserializeModel(string fileName)
        {
            SequentialModel model = null;
            if (File.Exists(fileName))
            {
                Stream stream = File.OpenRead(fileName);
                BinaryFormatter deserializer = new BinaryFormatter();
                model = (SequentialModel)deserializer.Deserialize(stream);
                stream.Close();
            }
            else
                throw new Exception("Trying to serialize a non-existing file.");

            return model;
        }
    }
}
