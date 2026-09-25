import { Component, type ErrorInfo, type ReactNode } from "react";
import { RefreshCw } from "lucide-react";

type ChatErrorBoundaryProps = {
  children: ReactNode;
  resetKey: string;
  onRetry: () => void;
};

type ChatErrorBoundaryState = {
  error: Error | null;
};

export class ChatErrorBoundary extends Component<
  ChatErrorBoundaryProps,
  ChatErrorBoundaryState
> {
  state: ChatErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ChatErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Chat runtime crashed", error, info);
  }

  componentDidUpdate(previous: ChatErrorBoundaryProps): void {
    if (previous.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  private retry = (): void => {
    this.setState({ error: null });
    this.props.onRetry();
  };

  render(): ReactNode {
    if (!this.state.error) return this.props.children;

    return (
      <div className="chat-crash-state" role="alert">
        <h2>聊天界面暂时无法显示</h2>
        <p>会话数据仍保留在服务端，可以重新加载后继续。</p>
        <button className="primary-button" type="button" onClick={this.retry}>
          <RefreshCw size={16} aria-hidden="true" />
          重新加载会话
        </button>
      </div>
    );
  }
}
