package com.example.Web.DTO;

import com.example.Web.Domain.User;
import com.example.Web.Domain.UserRole;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.validation.constraints.NotBlank;

@Getter
@Setter
@NoArgsConstructor
public class JoinRequest {

    @NotBlank(message = "로그인 아이디가 비어있습니다.")
    private String loginId;

    @NotBlank(message = "비밀번호가 비어있습니다.")
    private String password;
    private String passwordCheck;

    @NotBlank(message = "이름이 비어있습니다.")
    private String name;

    @NotBlank(message = "기기 아이디가 비어있습니다.")
    private Long deviceId;

    // 비밀번호 암호화 X
   /* public User toEntity() {
        return User.builder()
                .loginId(this.loginId)
                .password(this.password)
                .name(this.name)
                .role(UserRole.USER)
                .deviceId(this.deviceId)
                .build();
    }*/

    // 비밀번호 암호화
    public User toEntity(String encodedPassword) {
        return User.builder()
                .loginId(this.loginId)
                .password(encodedPassword)
                .name(this.name)
                .role(UserRole.USER)
                .deviceId(this.deviceId)
                .build();
    }
}