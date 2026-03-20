using NativeWebSocket;
using System;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public sealed class QRDetectionWebSocketClient
{
    [Serializable]
    struct DetectUnityRequest
    {
        public string action;
        public string image;
    }

    readonly WebSocket websocket;
    readonly Action<DetectionResponse> onDetectionResponse;

    bool sending;
    float lastSendTime;

    public QRDetectionWebSocketClient(string serverUrl, Action<DetectionResponse> onDetectionResponse)
    {
        websocket = new WebSocket(serverUrl);
        this.onDetectionResponse = onDetectionResponse;

        websocket.OnOpen += () => Debug.Log("QRDetectionRenderer: WebSocket connected");
        websocket.OnError += HandleSocketError;
        websocket.OnClose += HandleSocketClose;
        websocket.OnMessage += HandleMessage;
    }

    public Task ConnectAsync()
    {
        return websocket.Connect();
    }

    public void DispatchMessageQueue()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        websocket.DispatchMessageQueue();
#endif
    }

    public async void TrySend(Texture2D sourceTexture, float sendInterval)
    {
        if (sending)
            return;
        if (websocket.State != WebSocketState.Open)
            return;
        if (Time.time - lastSendTime < sendInterval)
            return;
        if (sourceTexture == null)
            return;

        sending = true;
        lastSendTime = Time.time;

        try
        {
            byte[] png = sourceTexture.EncodeToPNG();
            var request = new DetectUnityRequest
            {
                action = "detect_unity",
                image = Convert.ToBase64String(png),
            };
            string payload = JsonUtility.ToJson(request);
            await websocket.SendText(payload);
        }
        catch (Exception e)
        {
            sending = false;
            Debug.LogException(e);
        }
    }

    public async Task CloseAsync()
    {
        if (websocket.State == WebSocketState.Open)
            await websocket.Close();
    }

    void HandleMessage(byte[] bytes)
    {
        var json = Encoding.UTF8.GetString(bytes);
        var response = JsonUtility.FromJson<DetectionResponse>(json);

        if (!string.IsNullOrEmpty(response.error))
        {
            Debug.LogError("QRDetectionRenderer: Server error: " + response.error);
            sending = false;
            return;
        }

        sending = false;
        if (onDetectionResponse != null)
            onDetectionResponse(response);
    }

    void HandleSocketError(string error)
    {
        sending = false;
        Debug.LogError("QRDetectionRenderer: WebSocket error: " + error);
    }

    void HandleSocketClose(WebSocketCloseCode closeCode)
    {
        sending = false;
        Debug.Log("QRDetectionRenderer: WebSocket closed");
    }
}
